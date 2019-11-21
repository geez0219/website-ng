import { Component, OnInit } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { Title, Meta } from '@angular/platform-browser';

import { SnackbarComponent } from '../snackbar/snackbar.component';

@Component({
  selector: 'app-install',
  templateUrl: './install.component.html',
  styleUrls: ['./install.component.css']
})
export class InstallComponent implements OnInit {
  data = {
    name: 'Installing FastEstimator',
    bio: 'Installing FastEstimator',
  };

  durationInSeconds = 3;

  constructor(private _snackBar: MatSnackBar, private title: Title, private meta: Meta) { }

  ngOnInit() {
    this.title.setTitle(this.data.name);
    this.meta.addTags([
      { name: 'og:url', content: '/install' },
      { name: 'og:title', content: this.data.name },
      { name: 'og:description', content: this.data.bio },
    ]);
  }

  copied(event: any) {
    this._snackBar.openFromComponent(SnackbarComponent, {
      duration: this.durationInSeconds * 1000,
      data: 'Copied'
    });
  }
}
