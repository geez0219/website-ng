import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http'; 

import { NestedTreeControl } from '@angular/cdk/tree';
import { MatTreeNestedDataSource } from '@angular/material/tree';

import { API } from '../api';

@Component({
  selector: 'app-api',
  templateUrl: './api.component.html',
  styleUrls: ['./api.component.css']
})
export class ApiComponent implements OnInit {
  apiList: API;
  currentSelection: string;
  currentAPIText: string;

  treeControl = new NestedTreeControl<API>(node => node.children);
  dataSource = new MatTreeNestedDataSource<API>();

  constructor(private http: HttpClient) { }

  ngOnInit() {
    this.getAPIStructure();
    this.currentSelection = 'assets/api/Pipeline.md';
    this.getSelectedAPIText();
  }

  getAPIStructure() {
    this.http.get('assets/api/structure.json', {responseType: 'text'}).subscribe(data => {
      this.apiList = <API>JSON.parse(data);

      this.dataSource.data = this.apiList.children;
      // console.log(this.apiList);
    });
  }
  
  hasChild = (_: number, node: API) => !!node.children && node.children.length > 0;

  updateCurrentAPI(api: API) {
    this.currentSelection = 'assets/api/' + api.name;

    this.getSelectedAPIText();
  }

  getSelectedAPIText() {
    this.http.get(this.currentSelection, {responseType: 'text'}).subscribe(data => {
      this.currentAPIText = data;
    });
  }

}
