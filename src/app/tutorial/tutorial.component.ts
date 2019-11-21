import { Component, OnInit, OnDestroy, HostListener } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { ActivatedRoute, Router } from '@angular/router';
import { BehaviorSubject } from 'rxjs';

import { Tutorial } from '../tutorial';
import { TOC } from '../toc';

@Component({
  selector: 'app-tutorial',
  templateUrl: './tutorial.component.html',
  styleUrls: ['./tutorial.component.css']
})
export class TutorialComponent implements OnInit {
  selectedTutorial: string;
  currentTutorialText: string;
  tutorialList: Tutorial[];
  tocContent: TOC[];
  
  screenWidth: number;
  private screenWidth$ = new BehaviorSubject<number>(window.innerWidth);

  structureHeaderDict = {
    // 'Content-Type': 'application/json',
    'Accept': "application/json, text/plain",
    'Access-Control-Allow-Origin': '*'
  }
  structureRequestOptions = {
    headers: new HttpHeaders(this.structureHeaderDict),
  };

  contentHeaderDict = {
    'Accept': "application/json, text/plain",
    'Access-Control-Allow-Origin': '*'
  }
  contentHeaderOptions = {
    responseType: 'text' as 'text',
    headers: new HttpHeaders(this.contentHeaderDict)
  };


  @HostListener('window:resize', ['$event'])
  onResize(event) {
    this.screenWidth$.next(event.target.innerWidth);
  }

  constructor(private http: HttpClient, private router: Router, private route: ActivatedRoute) {
    this.route.params.subscribe(params => {
      if (params['name']) {
        this.selectedTutorial = params['name'];
        this.getTutorialStructure();
      }
    });

    this.screenWidth$.subscribe(width => {
      this.screenWidth = width;
    });
  }

  ngOnInit() {
  }

  getTutorialStructure() {
    this.http.get('assets/tutorial/structure.json', this.structureRequestOptions).subscribe(data => {
      this.tutorialList = <Tutorial[]>(data);

      var t: Tutorial[] = this.tutorialList.filter(tutorial => tutorial.name === (this.selectedTutorial + ".md"));
      this.updateTutorialContent(t[0]);
    });
  }

  updateTutorialContent(tutorial: Tutorial) {
    if (!tutorial)
      this.router.navigate(['PageNotFound']);
    
    window.scroll(0,0);

    this.getSelectedTutorialText('assets/tutorial/' + tutorial.name);
  }

  getSelectedTutorialText(tutorialName) {
    this.http.get(tutorialName, this.contentHeaderOptions).subscribe(data => {
      this.currentTutorialText = data;
    });
  }

}
